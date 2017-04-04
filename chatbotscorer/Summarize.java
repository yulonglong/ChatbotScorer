import java.util.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Summarize {
	TreeSet<String> folderNames;
	public Summarize() {
		folderNames = new TreeSet<String>();
	}

	public static String s_testFolderName = "../ed_clean_all_minus182_noMaybe_noLab";
	public static int s_startExptNum = 0;


	public void run() throws Exception{

		File folder = new File(s_testFolderName);
		File[] listofDirectories = folder.listFiles();

		int index = 0;
		for (int i = 0; i < listofDirectories.length; i++) {
			if (listofDirectories[i].isDirectory()) {
				folderNames.add(listofDirectories[i].getName());
			}
		}
		
		// Write Header
		System.out.println("ExptName\tMaj-f1\tMajAcc\ttest1-f1\ttest1-Acc\ttest2-f1\ttest2-Acc\ttest3-f1\ttest3-Acc\ttest4-f1\ttest4-Acc\ttest5-f1\ttest5-Acc\ttest-ave-f1\ttest-ave-Acc");
		
		String prevExptNum = "000";
		int countExpt = 0;
		Double totalTestF1 = 0.0;
		Double totalTestAcc = 0.0;
		// Go through each of the folder names
		for (String folderName: folderNames) {
			String exptNum = folderName.substring(4,7);
			if (Integer.parseInt(exptNum) < s_startExptNum) continue;
			System.err.println("Reading " + folderName + "...");
			System.err.println(exptNum);
			
			String majF1 = ""; String majAcc = "";
			String testF1 = ""; String testAcc = "";
			FileReader currentFile = new FileReader(s_testFolderName+"/"+folderName+"/log.txt");
			BufferedReader br = new BufferedReader(currentFile);
			String line;
			while ((line = br.readLine()) != null) {
				String majorityRegex = "(?i)\\[MAJ\\]  F1: ([0-9\\.]+), Acc: ([0-9\\.]+)";
				Pattern patternMajorityRegex = Pattern.compile(majorityRegex, Pattern.CASE_INSENSITIVE);
				Matcher mMajority = patternMajorityRegex.matcher(line);
				while (mMajority.find()) {
					majF1 = mMajority.group(1);
					majAcc = mMajority.group(2);
				}
				String testRegex = "(?i)\\[TEST\\] F1: ([0-9\\.]+), Acc: ([0-9\\.]+)";
				Pattern patternTestRegex = Pattern.compile(testRegex, Pattern.CASE_INSENSITIVE);
				Matcher mTest = patternTestRegex.matcher(line);
				while (mTest.find()) {
					testF1 = mTest.group(1);
					testAcc = mTest.group(2);
				}
			}
			if (!prevExptNum.equals(exptNum)) {
				if (countExpt > 0)
					System.out.println( String.format( "%.3f", totalTestF1/(double)countExpt) + "\t" + String.format( "%.3f", totalTestAcc/(double)countExpt));
				
				System.out.print(folderName + "\t");
				if (majF1.length() > 0 && majAcc.length() > 0) {
					System.out.print(majF1 + "\t" + majAcc + "\t");
				}
				prevExptNum = exptNum;
				countExpt = 0;
				totalTestF1 = 0.0;
				totalTestAcc = 0.0;
			}
			countExpt++;
			if (testF1.length() > 0 && testAcc.length() > 0) {
				System.out.print(testF1 + "\t" + testAcc + "\t");
				totalTestF1 += Double.parseDouble(testF1);
				totalTestAcc += Double.parseDouble(testAcc);
			}
		}
		System.out.println( String.format( "%.3f", totalTestF1/(double)countExpt) + "\t" + String.format( "%.3f", totalTestAcc/(double)countExpt));
	}

	public static void showErrorMessage() {
		System.err.println(
			"\n" +
			"=========== Summarize - HELP ============\n" +
			" This script is used to summarize the different NN runs/experiments.\n" +
			" In every folder, the results is taken from the last line in the log.txt file\n" +
			"=========================================\n"
		);
		System.err.println("Summarize: List of arguments available:");
		System.err.println("1.  -TestFolderName [The name of the Test folder]");
		System.err.println("      => (e.g. '-TestFolderName ../output_archive/')");
		System.err.println("      => List all folders in the specified directory and read each of the experiments");
		System.err.println("2.  -n [The number of files]");
		System.err.println("      => (e.g. '-n 4'");
		System.err.println("      => Summarize the experiments starting experiment n, in this example it summarizes from experiment n onwards");
		return;
	}

	public static void main(String[] args) {
		if (args.length > 0) {
			for(int i=0;i<args.length; i++) {
				if (args[i].equalsIgnoreCase("-testfoldername")) {
					if (i+1 >= args.length) {
						showErrorMessage();
						return;
					}
					else {
						s_testFolderName = args[i+1];
						i++; 
					}
				}
				else if (args[i].equalsIgnoreCase("-n")) {
					if (i+1 >= args.length) {
						showErrorMessage();
						return;
					}
					else {
						s_startExptNum = Integer.parseInt(args[i+1]);
						i++; 
					}
				}
				else {
					showErrorMessage();
					return;
				}
			}
		}
		else {
			showErrorMessage();
			return;
		}

		Summarize sf = new Summarize();
		try {
			sf.run();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
